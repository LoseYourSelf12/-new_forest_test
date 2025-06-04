import utils
import types
import os


class Config:
    _instance = None
    
    def __new__(class_, *args, **kwargs):
        if not isinstance(class_._instance, class_):
            class_._instance = object.__new__(class_, *args, **kwargs)
            class_.__keys = {}
        return class_._instance
    
    def load(self, file_name:str)->None:
        if file_name[-3:] == '.py':
            #загружаем модуль конфига
            module = utils.load_module(os.path.dirname(file_name), os.path.basename(file_name))
            for key in list(module.__dict__.keys())[7:]:
                value = module.__dict__[key]
                if not(isinstance(value, types.FunctionType) or 
                       isinstance(value, type) or 
                       isinstance(value, types.ModuleType)):
                    self.__keys[key] = value
        else:
            raise NotImplemented
        
    def append(self, key:str, value):
        self.__keys[key] = value
      
    def __getitem__(self, key:str):
        return self.__keys.get(key, None)

            