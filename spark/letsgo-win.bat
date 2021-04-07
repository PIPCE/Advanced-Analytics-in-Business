@echo off

echo Setting paths...

set CURRENT_HOME=%~dp0

IF NOT "%CURRENT_HOME%"=="%CURRENT_HOME: =%" (
    echo Uh-oh, you have spaces in your path
    echo Please move Spark to a location without spaces
    GOTO End
)

set SPARK_HOME=%CURRENT_HOME%spark-3.1.1-bin-hadoop2.7
set HADOOP_HOME=%CURRENT_HOME%winutils
set JAVA_HOME=%CURRENT_HOME%jre1.8.0_241
set PATH=%PATH%;%SPARK_HOME%\bin;%HADOOP_HOME%\bin;%JAVA_HOME%\bin
set PYSPARK_DRIVER_PYTHON=jupyter
set PYSPARK_DRIVER_PYTHON_OPTS='notebook'
set PYSPARK_PYTHON=python

:Java

echo Checking if Java exists...

java -version 1>NUL 2>NUL
IF NOT %ERRORLEVEL%==9009 GOTO Python

echo It doesn't -- try installing manually or adding java.exe to your path

GOTO End

:Python

echo Checking if Python exists...

python -V 1>NUL 2>NUL
IF NOT %ERRORLEVEL%==9009 GOTO Jupyter

echo Doesn't seem to be the case, I'll try to load in Anaconda

call C:\Users\%username%\Anaconda3\Scripts\activate.bat C:\Users\%username%\Anaconda3
call %localappdata%\Continuum\Anaconda3\Scripts\activate.bat %localappdata%\Continuum\Anaconda3
call C:\Users\%username%\miniconda3\Scripts\activate.bat C:\Users\%username%\miniconda3
            
python -V 1>NUL 2>NUL
IF NOT %ERRORLEVEL%==9009 GOTO Jupyter

echo Still nothing -- try to start this script from an Anaconda prompt

GOTO End

:Jupyter

echo Checking if Jupyter exists...

jupyter --version 1>NUL 2>NUL
IF NOT %ERRORLEVEL%==9009 GOTO Start

echo Looks like it doesn't -- did you install Jupyter in this environment?

GOTO End

:Start

echo Starting...

pyspark

:End

pause