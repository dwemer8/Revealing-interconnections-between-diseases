from typing import Any, Union, Dict, List, Optional

def get_type_conversion_function(type_name: str) -> Union[callable, None]:
    """
    Creates a function that converts input to the specified type.
    
    Args:
        type_name: Name of the type as string (e.g., 'int', 'str', 'List[int]')
        
    Returns:
        Conversion function or None if type not found
    """
    # Map type strings to actual types
    type_map = {
        'int': int,
        'str': str,
        'float': float,
        'bool': bool,
        'dict': dict,
        'list': list,
        'Any': Any,
        'Optional': Optional,
        'List': List,
        'Dict': Dict
    }
    
    # Get the actual type from the map
    type_ = type_map.get(type_name, None)
    if type_ is None:
        raise NotImplementedError(f"Type {type_name} not found, only {list(type_map.keys())} are supported")
        
    # Define the conversion function
    def conversion_function(value: Any):
        try:
            return type_(value)
        except ValueError:
            raise ValueError(f"Value {value} cannot be converted to type {type_name}")
    
    return conversion_function
