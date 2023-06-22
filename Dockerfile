FROM python:3.9
COPY . DependencyLengthSurvey
RUN pip install -r DependencyLengthSurvey/requirements.txt
CMD ["/bin/bash"]
