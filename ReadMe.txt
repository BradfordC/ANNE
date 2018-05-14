This project was tested on Python 3.6

--Prerequisites--
This project requires 2 python modules to be installed: numpy and PIL.

They can be installed with the following commands:

   pip install numpy
   pip install Pillow
   
--Setup--
The following is the folder format that the data should be in.

Data Folder/
--Category 1/
--Category 2/
--etc.

The program will use assign all images in each folder and its subfolders the same classification tag.
It should support most image formats.

Main.py should be edited so that LoadAllCategories() is given the path to "Data Folder/"
Main.py should be edited so that FileManager is pointed to a folder that actually exists, so it can save the results.
