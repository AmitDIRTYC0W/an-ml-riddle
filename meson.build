project('an-ml-riddle', 'cpp',
        license : 'MPL-2.0',
        version : '0.1.0')

subdir('include')
subdir('src')

inc = include_directories('include')

client_lib = library('anmlriddleclient', client_sources,
                     include_directories : inc,
                     version : meson.project_version(),
                     install : true)

server_lib = library('anmlriddleserver', server_sources,
                     include_directories : inc,
                     version : meson.project_version(),
                     install : true)

dep = declare_dependency(
        include_directories: inc,
        link_with : [client_lib, server_lib]
)

subdir('tests')
