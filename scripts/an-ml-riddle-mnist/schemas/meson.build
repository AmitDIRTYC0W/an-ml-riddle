# capnp = find_program('capnp')

# model_schema_header = custom_target('Model.capnp.*',
#                               output : ['Model.capnp.h', 'Model.capnp.c++'],
#                               input : 'Model.capnp',
#                               command : [capnp,
#                                          'compile',
#                                          '--src-prefix=@CURRENT_SOURCE_DIR@',
#                                          '--output=c++:@OUTDIR@',
#                                          '@INPUT@'],
#                               install : true,
#                               install_dir : [get_option('includedir') / 'anmlriddle', false],
#                               install_tag : ['devel'])

# messages_schema_header = custom_target('Messages.capnp.*',
#                               output : ['Messages.capnp.h', 'Messages.capnp.c++'],
#                               input : 'Messages.capnp',
#                               command : [capnp,
#                                          'compile',
#                                          '--src-prefix=@CURRENT_SOURCE_DIR@',
#                                          '--output=c++:@OUTDIR@',
#                                          '@INPUT@'],
#                               install : true,
#                               install_dir : [get_option('includedir') / 'anmlriddle', false],
#                               install_tag : ['devel'])

flatc = find_program('flatc')
flatc_gen = generator(flatc,
                      output : '@BASENAME@_generated.h',
                      arguments : ['--cpp',
                                   '-o',
                                   '@BUILD_DIR@',
                                   '@EXTRA_ARGS@',
                                   '@INPUT@'])
schema_sources = files('model.fbs', 'client_message.fbs', 'server_message.fbs', 'common.fbs')
schema_headers = flatc_gen.process(schema_sources, extra_args : ['--gen-object-api'])

# schema_headers = custom_target('amrschemas',
#                         # output : '@BASENAME@_generated.h',
#                         output : ['model_share_generated.h', 'vector_share_generated.h'], # TODO using a for loop
#                         input : schema_sources,
#                         command : [flatc,
#                                    '--cpp',
#                                    '-o',
#                                    '@OUTDIR@',
#                                    '@INPUT@'],
#                         install : true,
#                         install_dir : get_option('includedir') / 'amrschemas',
#                         install_tag : ['devel'])

schemas_dep = declare_dependency(
    # include_directories : [],
    sources : [schema_headers]
)
# libdistanceproto = library(meson.project_name(), distance_pb_cc,
#   dependencies: [protobuf],
#   install: true,
# )


# distanceproto_dep = declare_dependency(
#   link_with: libdistanceproto,
#   include_directories: [include_directories('.')],
#   dependencies: protobuf,
# )

# capnp = find_program('capnp')
# cp = find_program('cp')
# capnp_gen = generator(capnp,
#                       output : ['@PLAINNAME@.c++', '@PLAINNAME@.h'],
#                       arguments : ['compile',
#                                    '--src-prefix=' + meson.current_source_dir(),
#                                    '--output=c++:@BUILD_DIR@',
#                                    '@EXTRA_ARGS@',
#                                    '@INPUT@'])

# schemas = files('Model.capnp')
# schema_sources = capnp_gen.process(schemas)
# shared_sources += schema_sources

# foo_h = custom_target(
#     'Model.capnp.h',
#     output : 'Model.capnp.h',
#     command : [cp, 'Model.capnp.h', '@BUILD_DIR@/include/amrc'],
# )
