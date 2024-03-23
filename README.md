# ECSE552-project

## Setting up a local repo
1. Change the current working directory to your local project. 
This can be done running the following command:
```
cd [your_project_directory]
```
2. Initialize as a git repository.
```
git init
```
4. Configure the origin as a remote repository.
Run the following command (we name the remote repository upstream):
```
git remote add upstream https://github.com/tanakaakiyama/Temperature-Characterization-System-for-Biosensing-Devices.git
```
5. Sync your local repository with the upstream
```
git fetch upstream
```
6. Merge
```
git merge upstream/master
```

## Pushing changes to remote repository
To push changes, first stage all changed files by running the following command:
```
git add .
```
Then commit and add a message:
```
git commit -m "Add your message here"
```
Then push your changes to the remote repository
```
git push upstream master
```
