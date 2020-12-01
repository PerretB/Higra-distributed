set(TEST_FILES
        test_extension.py)

foreach (FILE ${TEST_FILES})
    configure_file(${FILE} ${CMAKE_CURRENT_BINARY_DIR}/${FILE} COPYONLY)
endforeach ()
        
add_custom_target(all_tests ALL
        DEPENDS higra_distributed
        )

        
if (DO_AUTO_TEST)
    if (WIN32)
        set (PYTHON_TEST_COMMAND "import sys;  sys.path.insert(0, '${CMAKE_LIBRARY_OUTPUT_DIRECTORY}');  import unittest;  result=unittest.TextTestRunner().run(unittest.defaultTestLoader.discover('${CMAKE_CURRENT_BINARY_DIR}'));  exit(0 if result.wasSuccessful() else 1)")
        add_custom_command(TARGET all_tests
            COMMENT "Run tests"
            POST_BUILD COMMAND ${PYTHON_EXECUTABLE} -c \"${PYTHON_TEST_COMMAND}\"
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                )
    else()
        add_custom_command(TARGET all_tests
            COMMENT "Run tests"
            POST_BUILD COMMAND ${PYTHON_EXECUTABLE} -c \"import sys $<SEMICOLON>  sys.path.insert(0, '${CMAKE_BINARY_DIR}/src') $<SEMICOLON>  import unittest $<SEMICOLON>  result=unittest.TextTestRunner() .run(unittest.defaultTestLoader.discover('${CMAKE_CURRENT_BINARY_DIR}')) $<SEMICOLON>  exit(0 if result.wasSuccessful() else 1) \"
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                )
    endif()
endif ()
