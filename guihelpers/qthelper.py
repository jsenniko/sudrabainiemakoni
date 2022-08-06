from PyQt5.QtWidgets import QFileDialog, QInputDialog

def gui_fname(directory=None, caption="", filter=""):
    """Select a file via a dialog and return the file name."""
    if directory is None: directory ='./'
    fname = QFileDialog.getOpenFileName(None, caption,
                directory, filter=filter)
    return fname[0]
def gui_dir(directory=None, caption=""):
    """Select a file via a dialog and return the file name."""
    if directory is None: directory ='./'
    fname = QFileDialog.getExistingDirectory(None, caption=caption, directory=directory)
    return fname
def gui_string(name="",caption=""):
    res, Ok = QInputDialog.getText(None, name, caption)
    return res if Ok else None
