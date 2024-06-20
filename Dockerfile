# Use an official Python runtime as a parent image
FROM python:latest

# Set the working directory in the container
WORKDIR /pages

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential gcc gfortran libatlas-base-dev && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /pages
COPY . /pages

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variable
ENV PORT 8501

# Run app.py when the container launches
CMD ["streamlit", "run", "Home.py"]
