FROM python:3.9
COPY . DependencyLengthSurvey
RUN pip install -r requirements.txt
CMD ["/bin/bash"]
