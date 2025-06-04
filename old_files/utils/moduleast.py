import ast
import types
import os


# динамически загружает модуль по имени и пути
def load_module(path:str, name:str, global_vars=None)->types.ModuleType:
    """
        Динамически загружает модуль по имени файла и пути к нему
    """

    # Чтение исходного кода из файла
    file_path = os.path.join(path, name)
    with open(file_path, 'r', encoding='utf-8') as file:
        source_code = file.read()

    # Парсинг исходного кода в AST    
    tree = ast.parse(source_code, filename=file_path)

    # Создание нового модуля
    module = types.ModuleType(name)
    module.__file__ = file_path

    # Компиляция дерева AST в исполняемый объект
    compiled_code = compile(tree, filename=file_path, mode='exec')
    
    # Обновление пространства имен модуля глобальными переменными
    if global_vars is not None:
        module.__dict__.update(global_vars)

    # Исполнение кода в пространстве имен модуля
    exec(compiled_code, module.__dict__)

    return module
