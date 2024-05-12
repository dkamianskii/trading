import shutil
from os import path, makedirs


def create_sub_fld_if_not_exist(fld_path: str, sub_fld_name: str) -> str:
    sub_fld_path = path.join(fld_path, sub_fld_name)
    if not path.exists(sub_fld_path):
        makedirs(sub_fld_path)
    return sub_fld_path


def create_empty_sub_fld(fld_path: str, sub_fld_name: str) -> str:
    sub_fld_path = path.join(fld_path, sub_fld_name)
    if path.exists(sub_fld_path):
        shutil.rmtree(sub_fld_path)
    makedirs(sub_fld_path)
    return sub_fld_path