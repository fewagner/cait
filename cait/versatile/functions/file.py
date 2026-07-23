from typing import List
       
def combine(fname: str, 
            files: List[str], 
            src_dir: str = '', 
            out_dir: str = '', 
            groups_combine: List[str] = ["events", "testpulses", "noise"], 
            groups_include: List[str] = [],
            extend_hours: bool = True
            ):
    raise NotImplementedError("This function was moved to 'cait.data.combine_h5'.")
    
def merge(fname: str, 
          files: List[str], 
          src_dir: str = '', 
          out_dir: str = '', 
          groups_merge: List[str] = ["events", "testpulses", "noise"], 
          groups_include: List[str] = [],
          extend_hours: bool = True):
    
    raise NotImplementedError("This function was moved to 'cait.data.merge_h5'.")