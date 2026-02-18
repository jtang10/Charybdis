configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(CHARYBDIS_TEST_DEPENDS
  charybdis-opt
  FileCheck
)

add_lit_testsuite(check-charybdis "Running Charybdis FileCheck tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CHARYBDIS_TEST_DEPENDS}
)

set_target_properties(check-charybdis PROPERTIES FOLDER "Tests")
