FROM openai/retro-agent:pytorch
ADD a3c-docker.py .
ADD greenhillzone.act1/ greenhillzone.act1/
RUN . ~/venv/bin/activate && pip install scipy tensorboardX
WORKDIR /root/compo/
CMD ["python", "/root/compo/a3c-docker.py", "--env greenhillzone.act2", "--sonic True", "--docker True"]