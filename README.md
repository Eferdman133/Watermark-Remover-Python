This is a standalone application that takes a pdf input and removes all watermarks from the document before creating a separate copy with "(No Watermarks) " prepended to the front. (the original file is preserved). Features include:

1. Drag & drop
2. Before & after file preview
3. Support for multiple files at once (without preview, but the option to open all processed documents simultaneously after program is ran)
4. Lightning fast speed, even for larger documents

The .exe should be a runnable standalone program. When you run the program, you will get a pop-up saying "Windows Protected Your PC." You will have to press on the "More Info" button, then "Run Anyway." If the .exe runs too slowly, or Windows flags the programs as a false positive, you will need to install Python:

1. Open the Microsoft Store app by pressing the search bar in the task bar -> type in "Microsoft Store" -> open the app
2. In the app, search for "Python" and click on the most recent version (Python 3.13 as of this readme)
3. Press "Get"
4. Once the application downloads, press Windows Key + R -> type "CMD" (without the quotes) -> press enter (alternatively, press the search bar in the task bar and look for 5. "CMD" -> press on "Command Prompt")
6. When the command line opens, type (without quotations) "Python -m pip install pymupdf PyQt6 Pillow" (this should install all the necessary dependencies) and press enter
7. run the program by double clicking on WatermarkRemover.py

You may receive an error when trying to install some of the dependencies (potentially a long path error). If this happens, I have included a 'lite' version that only uses the PyMuPDF dependency for the script. If you have followed the above steps, check if you have the necessary dependencies:

1. Open up the Command Prompt (press Windows Key + R -> type "CMD" (without the quotes) -> press enter or, alternatively, press the search bar in the task bar and look for "CMD" -> press on "Command Prompt")
2. In the command line, type "Python -m pip show pymupdf" and press enter

If you see information about the package, it has been installed correctly. If instead you see "WARNING: Package(s) not found: pymupdf" proceed to the next step

3. In the command line, type "Python -m pip install pymupdf"
4. Confirm the package was installed by repeating step 2

If you see information about the PyMuPDF library, you should now be able to double click the "WatermarkRemoverLite.py" file and have it open up. Some features may be unavailable, should as the drag & drop and file preview, but the core functionality should be preserved. Please contact me if issues persist or if you find any bugs/unwanted removals/suggestions.
