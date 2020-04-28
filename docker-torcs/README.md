# Docker-TORCS

TORCS running in headless mode on Docker with a `flask` and `ray` interface. Currently a lot of the TORCS code is commented out in order to get a working dockerized Ray cluster on a simple problem. Feel free to uncomment and fix everything if you wish.

## Setup

* Install Docker
* Clone this repository
* Run `automator.py`
* The orchestrator is present in the `Orchestrator` folder.


## Additional Notes

* `Dockerfile` and `Dockerfile_orchestrator` are example configs of the dockerfiles that will be used.
* This uses a custom patch of `xvfbwrapper` and is therefore defined as a separate repository in `requirements.txt`
