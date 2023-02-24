### How to run?

1. Install docker if you don't have it
2. Run
	```
	docker-compose up --build (to build it)
    docker-compose up (to just run)
	```
### How to test it?
1. Once you run the above commands, go to browser and then paste this url
	```
		http://localhost:8073
	```
2. Start uploading sample files. You can get sample files from here:
	``` 
    /samples/
		1.jpg
		2.jpg
   
    ```
### Github action CI/CD implementation
Everytime achange is pushed to GIT repository, it's going to build and publish a docker image for the application using GitHub actions.


