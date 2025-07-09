#! /bin/bash
# Activation script

# Remove flags setup from cxx-compiler
unset CFLAGS
unset CPPFLAGS
unset CXXFLAGS
unset DEBUG_CFLAGS
unset DEBUG_CPPFLAGS
unset DEBUG_CXXFLAGS
unset LDFLAGS

if [[ $host_alias == *"apple"* ]];
then
  # On OSX setting the rpath and -L it's important to use the conda libc++ instead of the system one.
  # If conda-forge use install_name_tool to package some libs, -headerpad_max_install_names is then mandatory
  export LDFLAGS="-Wl,-headerpad_max_install_names -Wl,-rpath,$CONDA_PREFIX/lib -L$CONDA_PREFIX/lib"
elif [[ $host_alias == *"linux"* ]];
then
  # On GNU/Linux, I don't know if these flags are mandatory with g++ but
  # it allow to use clang++ as compiler
  export LDFLAGS="-Wl,-rpath,$CONDA_PREFIX/lib -Wl,-rpath-link,$CONDA_PREFIX/lib -L$CONDA_PREFIX/lib"
fi

# Setup ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache

# Create compile_commands.json for language server
export CMAKE_EXPORT_COMPILE_COMMANDS=1

# Activate color output with Ninja
export CMAKE_COLOR_DIAGNOSTICS=1

# Set default build value only if not previously set
export NANOEIGENPY_BUILD_TYPE=${NANOEIGENPY_BUILD_TYPE:=Release}
export NANOEIGENPY_CHOLMOD_SUPPORT=${NANOEIGENPY_CHOLMOD_SUPPORT:=OFF}
export NANOEIGENPY_ACCELERATE_SUPPORT=${NANOEIGENPY_ACCELERATE_SUPPORT:=OFF}
