import subprocess
import os # TODO
import pathlib

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


schemas = ['schemas/']


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        root = pathlib.Path(self.root)
        schemas_dir = root / 'schemas'
        # print(list(schemas_dir.iterdir()))
        
        schemas = [schemas_dir / 'common.fbs', schemas_dir / 'model.fbs']
        subprocess.run(['flatc', '--python', '-o', root / 'anmlriddle_mnist', *schemas], check=True)
        # exit(3)
