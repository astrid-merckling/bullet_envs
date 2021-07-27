
# git add . *
# git commit -m "$1"
# git pull
# git push

set -e
set -x

git pull
git add --all
git commit -m "$@"
git push origin master
