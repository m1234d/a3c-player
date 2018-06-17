from __future__ import print_function
from retro_contest.local import make
import torch, os, gym, time, glob, argparse, sys
import numpy as np
from scipy.signal import lfilter
from scipy.misc import imresize # preserves single-pixel info _unlike_ img = img[::2,::2]
from gym import spaces
import torch.nn as nn
import cv2
import torch.nn.functional as F
import torch.multiprocessing as mp
import tensorboardX
from matplotlib import pyplot as pyplot
from tensorboardX import SummaryWriter
from baselines.common.atari_wrappers import FrameStack
os.environ['OMP_NUM_THREADS'] = '1'

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', default='Pong-v4', type=str, help='gym environment')
    parser.add_argument('--processes', default=mp.cpu_count(), type=int, help='number of processes to train with')
    parser.add_argument('--render', default=False, type=bool, help='renders the atari environment')
    parser.add_argument('--test', default=False, type=bool, help='sets lr=0, chooses most likely actions')
    parser.add_argument('--rnn_steps', default=20, type=int, help='steps to train LSTM over')
    parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
    parser.add_argument('--seed', default=1, type=int, help='seed random # generators (for reproducibility)')
    parser.add_argument('--gamma', default=0.99, type=float, help='rewards discount factor')
    parser.add_argument('--tau', default=1.0, type=float, help='generalized advantage estimation discount')
    parser.add_argument('--horizon', default=0.99, type=float, help='horizon for running averages')
    parser.add_argument('--hidden', default=256, type=int, help='hidden size of GRU')
    parser.add_argument('--sonic', default=False, type=bool, help='sonic')
    return parser.parse_args()
    
args = get_args()
args.save_dir = '{}/'.format(args.env.lower()) # keep the directory structure simple
writer = SummaryWriter("./" + args.save_dir + "/runs")

discount = lambda x, gamma: lfilter([1],[1,-gamma],x[::-1])[::-1] # discounted rewards one liner

#can be multiples of 16 starting at 80 (80, 96, 112, 128, etc.)
state_width = 80
state_height = 80

#prepro = lambda img: imresize(img[35:195].mean(2), (80,80)).astype(np.float32).reshape(1,80,80)/255.

def prepro(img):
    return img
    
def printlog(args, s, end='\n', mode='a'):
    print(s, end=end) ; #f=open(args.save_dir+'log.txt',mode) ; f.write(s+'\n') ; f.close()
    
def make_env(game, state=None, stack=False, scale_rew=False):
    """
    Create an environment with some standard wrappers.
    """
    if(state==None):
        env = gym.make(game)
    else: env = make(game=game, state=state); env = SonicDiscretizer(env); #env = AllowBacktracking(env) #CODE RUNNING HAS THIS ENABLED
    if scale_rew:
        env = RewardScaler(env)
    env = WarpFrame(env)
    if stack:
        env = FrameStack(env, 4)
    return env

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = state_width
        self.height = state_height
        self.observation_space = spaces.Box(low=0, high=255,
            shape=(1, self.height, self.width), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)/255.0
        return frame[:, :, None]

class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()

class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.
    This is incredibly important and effects performance
    drastically.
    """
    def reward(self, reward):
        return reward * 0.01

class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """
    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self, **kwargs): # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action): # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info
        
class NNPolicy(nn.Module): # an actor-critic neural network
    def __init__(self, channels, memsize, num_actions):
        super(NNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.gru = nn.GRUCell(32 * 5 * 5* state_width/80 * state_height/80, memsize)
        self.critic_linear, self.actor_linear = nn.Linear(memsize, 1), nn.Linear(memsize, num_actions)

    def forward(self, inputs, train=True, hard=False):
        inputs, hx = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        #print(x.norm())
        #print(hx.norm())
        #print(x.shape)
        hx = self.gru(x.view(-1, 32 * 5 * 5 * state_width/80 * state_height/80), (hx))
        #print(hx.norm())
        return self.critic_linear(hx), self.actor_linear(hx), hx

    def try_load(self, save_dir):
        paths = glob.glob(save_dir + '*.tar') ; step = 0
        if len(paths) > 0:
            ckpts = [int(s.split('.')[-2]) for s in paths]
            ix = np.argmax(ckpts) ; step = ckpts[ix]
            self.load_state_dict(torch.load(paths[ix]))
        print("\tno saved models") if step is 0 else print("\tloaded model: {}".format(paths[ix]))
        return step

class SharedAdam(torch.optim.Adam): # extend a pytorch optimizer so it shares grads across processes
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['shared_steps'], state['step'] = torch.zeros(1).share_memory_(), 0
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                
        def step(self, closure=None):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    self.state[p]['shared_steps'] += 1
                    self.state[p]['step'] = self.state[p]['shared_steps'][0] - 1 # a "step += 1"  comes later
            super.step(closure)

def cost_func(args, values, logps, actions, rewards, rank):
    np_values = values.view(-1).data.numpy()
    #np_values[-1] = bootstrap value
    #np_values is values_plus
    #rewards is rewards_plus
    # generalized advantage estimation using \delta_t residuals (a policy gradient method)
    delta_t = np.asarray(rewards) + args.gamma * np_values[1:] - np_values[:-1] #advantage
    logpys = logps.gather(1, torch.tensor(actions).view(-1,1))

    gen_adv_est = discount(delta_t, args.gamma * args.tau) #final discounted advantage
    policy_loss = -(logpys.view(-1) * torch.FloatTensor(gen_adv_est.copy())).sum() #negative sum of advantage * log-probabilities
    
    # l2 loss over value estimator
    rewards[-1] += args.gamma * np_values[-1] #calculate discounted value 
    discounted_r = discount(np.asarray(rewards), args.gamma)
    discounted_r = torch.tensor(discounted_r.copy(), dtype=torch.float32)
    value_loss = .5 * (discounted_r - values[:-1,0]).pow(2).sum() #mean-squared error
    #print(rewards)
    #print(discounted_r)
    entropy_loss = -(-logps * torch.exp(logps)).sum() # sum of probabilities * logprobabilities : encourage lower entropy
    return policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

def train(shared_model, shared_optimizer, rank, args, info):
    if args.sonic: env = make_env(game='SonicTheHedgehog-Genesis', state=args.env, stack=False, scale_rew=False)
    else: env = make_env(args.env) # make a local (unshared) environment
    env.seed(args.seed + rank) ; torch.manual_seed(args.seed + rank) # seed everything
    model = NNPolicy(channels=1, memsize=args.hidden, num_actions=args.num_actions) # a local/unshared model
    state = torch.tensor(prepro(env.reset())).type(torch.FloatTensor) # get first state
    testImg = state.numpy()[:, :, 0]
    start_time = last_disp_time = last_disp_time_2 = time.time()
    episode_length, epr, eploss, done  = 0, 0, 0, True # bookkeeping
    measure_counter = 0
    bestReward = 0
    lostProgress = 0
    print(str(rank) + "initialized")
    while info['frames'][0] <= 8e7 or args.test: # openai baselines uses 40M frames...we'll use 80M
        model.load_state_dict(shared_model.state_dict()) # sync with shared model

        hx = torch.zeros(1, 256) if done else hx.detach()  # rnn activation vector
        values, logps, actions, rewards = [], [], [], [] # save values for computing gradientss
        
        for step in range(args.rnn_steps):
            #print(state.shape)
            #img = state.numpy()
            #img = img.squeeze(2)
            #pyplot.imshow(img, cmap="gray")
            #pyplot.show()
            episode_length += 1

            value, logit, hx = model((state.view(1,1,state_width,state_height).type(torch.FloatTensor), hx))
            logp = F.log_softmax(logit, dim=-1)
            action = torch.exp(logp).multinomial(num_samples=1).data[0]#logp.max(1)[1].data if args.test else
            #if args.render:
                #action = logp.max(1)[1].data
            if(args.sonic): 
                a = action.numpy()[0]
                b = np.zeros(args.num_actions, dtype=int)
                b[a] = 1
                if rank == 0:
                    #print(state)
                    #print(logp)
                    pass
                state, reward, done, _ = env.step(action.numpy()[0])
                #if rank == 0:
                #    print(reward)
            else:
                state, reward, done, _ = env.step(action.numpy()[0])
            if args.render: env.render()

            state = torch.tensor(prepro(state)).type(torch.FloatTensor) ; epr += reward
            
            #if(epr > bestReward):
            #    bestReward = epr
            #    lostProgress = 0
            #else:
            #    lostProgress += 1

            #if lostProgress > 200:
            #    lostProgress = 0
            #    done = True
                
            #print(reward)
            if args.sonic:
                #reward = np.clip(reward, -1, 1) # TEST THIS, TRY CHANGING -1 to 0 OR 2 BACK TO 1 AND THE NEW CODE COMMENTED BELOW
                reward = reward * 0.01
            else:
                reward = np.clip(reward, -1, 1)
            done = done or episode_length >= 1e4 # don't playing one ep for too long
            
            info['frames'].add_(1) ; num_frames = int(info['frames'].item())
            if num_frames % 1e5 == 0: # save every 2M frames
                printlog(args, '\n\t{:.0f}M frames: saved model\n'.format(num_frames/1e6))
                torch.save(shared_model.state_dict(), args.save_dir+'model.{:.0f}.tar'.format(num_frames/1e5))
            
            if done: # update shared data
                if rank == 0:
                    writer.add_scalar("rewards/reward_" + str(rank), epr, info['frames'])
                    writer.add_scalar("losses/loss_" + str(rank), eploss, info['frames'])
                    writer.add_scalar("rewards/running_reward", info['run_epr'].item(), info['frames'])
                    writer.add_scalar("losses/running_loss", info['run_loss'].item(), info['frames'])
                    writer.add_scalar("rewards/value", str(value[0][0].item()), info['frames'])
                    time.sleep(1)
                    printlog(args, "Data written for " + str(rank))
                    printlog(args, "Reward: " + str(epr))
                info['episodes'] += 1
                interp = 1 if info['episodes'][0] == 1 else 1 - args.horizon
                info['run_epr'].mul_(1-interp).add_(interp * epr)
                info['run_loss'].mul_(1-interp).add_(interp * eploss)
                
            if rank == 0 and time.time() - last_disp_time_2 > 1:
                printlog(args, '{:.0f} f/s'.format(num_frames - measure_counter))
                measure_counter = int(info['frames'].item())
                last_disp_time_2 = time.time()
                
            if rank == 0 and time.time() - last_disp_time > 60: # print info ~ every minute
                elapsed = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time))
                printlog(args, 'time {}, episodes {:.0f}, frames {:.1f}M, mean epr {:.2f}, run loss {:.2f}'
                    .format(elapsed, info['episodes'].item(), num_frames/1e6,
                    info['run_epr'].item(), info['run_loss'].item()))
                last_disp_time = time.time()

            if done: # maybe print info.
                episode_length, epr, eploss, bestReward = 0, 0, 0, 0
                state = torch.tensor(prepro(env.reset()))

            values.append(value) ; logps.append(logp) ; actions.append(action) ; rewards.append(reward)
            if done: #NEW TEST THIS
                break #NEW TEST THIS
        #next_value = torch.zeros(1,1) if done else model((state.unsqueeze(0).view(1, 1, state_width, state_height), hx))[0]
        #print(torch.zeros(1, 1))
        #print(torch.from_numpy(np.array([[epr]])).type(torch.FloatTensor) )
        if args.sonic:
            eprr = epr*0.01
        else:
            eprr = epr
        next_value = torch.from_numpy(np.array([[pow(eprr, 2)]])).type(torch.FloatTensor) if done else model((state.unsqueeze(0).view(1, 1, state_width, state_height), hx))[0]
        values.append(next_value.detach())
        

        loss = cost_func(args, torch.cat(values), torch.cat(logps), torch.cat(actions), np.asarray(rewards), rank)
        eploss += loss.item()

        if(args.lr != 0):
            shared_optimizer.zero_grad() ; loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 40)
    
            for param, shared_param in zip(model.parameters(), shared_model.parameters()):
                if shared_param.grad is None: shared_param._grad = param.grad # sync gradients with shared model
            shared_optimizer.step()
            
if __name__ == "__main__":
    if sys.version_info[0] > 2:
        mp.set_start_method('spawn', force=True) # this must not be in global scope
    elif sys.platform == 'linux' or sys.platform == 'linux2':
        raise "Must be using Python 3 with linux!" # or else you get a deadlock in conv2d
    
    args = get_args()
    args.save_dir = '{}/'.format(args.env.lower()) # keep the directory structure simple
    if args.render:  args.processes = 1 ; args.test = True # render mode -> test mode w one process
    if args.test:  args.lr = 0 # don't train in render mode
    
    if args.sonic: args.num_actions = make_env(game='SonicTheHedgehog-Genesis', state=args.env, stack=False, scale_rew=False).action_space.n # get the action space of this game
    else: args.num_actions = make_env(args.env).action_space.n 
    
    print(args.num_actions)
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None # make dir to save models etc.

    torch.manual_seed(args.seed)
    shared_model = NNPolicy(channels=1, memsize=args.hidden, num_actions=args.num_actions).share_memory()
    shared_optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)

    info = {k: torch.DoubleTensor([0]).share_memory_() for k in ['run_epr', 'run_loss', 'episodes', 'frames']}
    info['frames'] += shared_model.try_load(args.save_dir) * 1e5
    if int(info['frames'].item()) == 0: printlog(args,'', end='', mode='w') # clear log file
    processes = []
    for rank in range(args.processes):
        p = mp.Process(target=train, args=(shared_model, shared_optimizer, rank, args, info))
        p.start() ; processes.append(p)
    for p in processes: p.join()
    writer.close()