
git log
	logging history of git repo
git log -n 5
	last 5 commiting history
git log --oneline
	remove the extra space 
git log --stat
	advance log history, it will show the no of lines added and deleted.
git log -p
	it shows the patch of each commit, as the diff of each commit
git log --graph --decorate --oneline
	it shows the graph (helpful while have more than one branches)


## Undoing Changes

git checkout commit_a13944
	It will checkout to that commit, but with detached head. **Note: Any commit you made, will be ignored, this is whetre conflict occurs. So solution is following command.**
git checkout -b new_branch_with_head_detached
	This will create a new branch and create a new timeline history from that branch on.

git reset --hard commit_a135578
	This will delete the commit after commit_a135578 from the local, but will cause complication with the remote repo.

git revert HEAD
	 A revert operation will take the specified commit, inverse the changes from that commit, and create a new "revert commit". 

- Use git checkout to move around and review the commit history
- git revert is the best tool for undoing shared public changes
- git reset is best used for undoing local private changes

#### git revert vs reset
git revert make a new commit using the old specified commit, where as git reset, delete the all commit in between the current and specified commit, it will delete the complete history for those commits(not a good idea, whuiile working in collaborations)


git revert commit_a135578
git revert HEAD~3
    Revert the changes specified by the fourth last commit in HEAD and create a new commit with the reverted changes.
git revert -n HEAD~3
    -n flag is for not committing messeage.
