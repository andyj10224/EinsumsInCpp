include_guard()

# If MKL is found then we must use it.
if (TARGET MKL::MKL)
    if (NOT EINSUMS_FFT_LIBRARY MATCHES MKL)
        # Do this atleast until we can ensure we link to FFTW3 for FFT and not MKL.
        message(FATAL_ERROR "MKL was detected. You must use MKL's FFT library")
    endif()
endif()

if (EINSUMS_FFT_LIBRARY MATCHES MKL)
    if (TARGET MKL::MKL)
        add_library(FFT::FFT ALIAS MKL::MKL)
    else()
        message(FATAL_ERROR "MKL FFT library library requested but MKL not found.")
    endif()
elseif(EINSUMS_FFT_LIBRARY MATCHES FFTW3)
    set(FFTW_USE_STATIC_LIBS TRUE)

    # Attempt to find FFTW for real
    find_package(FFTW
        COMPONENTS
            FLOAT_LIB
            DOUBLE_LIB
    )

    if (NOT FFTW_FLOAT_LIB_FOUND OR NOT FFTW_DOUBLE_LIB_FOUND)
        message(FATAL_ERROR "Did not find FFTW3.")
    else()
        add_library(FFT::FFT INTERFACE IMPORTED)
        target_link_libraries(FFT::FFT
            INTERFACE
                FFTW::Float
                FFTW::Double
        )
    endif()

    # If not found, check to see if MKL is available
    # If MKL is available then use MKL's implementation of FFTW.

    # Needs to be tested: If MKL and separate FFTW are found and separate
    # FFTW is requested...what happens?

endif()

# Make sure an FFT library was found
if (NOT TARGET FFT::FFT)
    message(FATAL_ERROR "No FFT library was found or being built.")
endif()