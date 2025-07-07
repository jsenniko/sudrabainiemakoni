from PyQt5.QtWidgets import QFileDialog, QInputDialog, QMessageBox
import os

class FileDialogManager:
    """
    A singleton class to manage file dialogs and remember the last used directory
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FileDialogManager, cls).__new__(cls)
            cls._instance.last_directory = os.path.dirname(os.path.abspath(__file__))
        return cls._instance
    
    def get_open_filename(self, directory=None, caption="", filter=""):
        """Select a file via a dialog and return the file name."""
        if directory is None:
            directory = self.last_directory
        elif os.path.isfile(directory):
            # If a file path was passed, use its directory
            directory = os.path.dirname(directory)
            
        fname = QFileDialog.getOpenFileName(None, caption, directory, filter=filter)
        
        if fname[0]:
            # Update the last used directory
            self.last_directory = os.path.dirname(os.path.abspath(fname[0]))
        
        return fname[0]
    
    def get_save_filename(self, directory=None, caption="", filter=""):
        """Select a file to save via a dialog and return the file name."""
        if directory is None:
            directory = self.last_directory
        elif os.path.isfile(directory):
            # If a file path was passed, use it directly
            pass
            
        fname = QFileDialog.getSaveFileName(None, caption, directory, filter=filter)
        
        if fname[0]:
            # Update the last used directory
            self.last_directory = os.path.dirname(os.path.abspath(fname[0]))
        
        return fname[0]
    
    def get_existing_directory(self, directory=None, caption=""):
        """Select a directory via a dialog and return the directory name."""
        if directory is None:
            directory = self.last_directory
            
        fname = QFileDialog.getExistingDirectory(None, caption=caption, directory=directory)
        
        if fname:
            # Update the last used directory
            self.last_directory = os.path.abspath(fname)
        
        return fname

# Create a global instance
file_dialog_manager = FileDialogManager()

def gui_fname(directory=None, caption="", filter=""):
    """Select a file via a dialog and return the file name."""
    return file_dialog_manager.get_open_filename(directory, caption, filter)

def gui_save_fname(directory=None, caption="", filter=""):
    """Select a file to save via a dialog and return the file name."""
    return file_dialog_manager.get_save_filename(directory, caption, filter)

def gui_dir(directory=None, caption=""):
    """Select a directory via a dialog and return the directory name."""
    return file_dialog_manager.get_existing_directory(directory, caption)

def gui_string(text="", name="", caption=""):
    res, Ok = QInputDialog.getText(None, name, caption, text=text)
    return res if Ok else None

def gui_confirm(caption="Confirm", title="Confirmation"):
    """Show a confirmation dialog with Yes/No buttons."""
    reply = QMessageBox.question(None, title, caption, 
                                QMessageBox.Yes | QMessageBox.No, 
                                QMessageBox.No)
    return reply == QMessageBox.Yes