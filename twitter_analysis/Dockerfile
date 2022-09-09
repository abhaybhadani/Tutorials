FROM python:3.9

RUN useradd -m app_user && apt-get update && apt-get -y install vim sudo
RUN  echo "ALL  ALL = (ALL) NOPASSWD: ALL" > /etc/sudoers


COPY . /home/app_user/app
RUN chown app_user.app_user /home/app_user -R
WORKDIR /home/app_user/app/src
RUN pip install --upgrade pip setuptools && pip install -r requirements.txt

USER app_user
EXPOSE 5000

ENTRYPOINT ["python","flask_app.py"]

