import os
from git import Repo

if not os.path.exists('mnist-pngs'):
    Repo.clone_from("https://github.com/rasbt/mnist-pngs", "mnist-pngs")