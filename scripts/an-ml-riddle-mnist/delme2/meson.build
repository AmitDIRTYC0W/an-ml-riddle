project('anmlriddle-mnist',
        license : 'MIT',
        version : '0.1.0')

py = import('python').find_installation(pure : true)

# py.install_sources([''])
subdir('anmlriddle_mnist')

py.extension_module('anmlriddle-mnist')
py.install_sources(anmlriddle_mnist_sources,
                   subdir : 'anmlriddle-mnist')
