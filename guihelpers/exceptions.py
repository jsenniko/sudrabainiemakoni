import functools
import sys
import inspect


def handle_exceptions(method_name=None):
    """
    A decorator that catches and handles exceptions from GUI methods.
    
    Can be used in two ways:
    1. @handle_exceptions - no arguments 
    2. @handle_exceptions("Custom method name") - with a custom name
    
    Args:
        method_name: Either the function being decorated (when used without parentheses)
                      or a string with a custom method name (when used with parentheses)
    
    Returns:
        The decorated function with exception handling.
    """
    def decorator(func):
        # Determine display name based on decorator usage
        if isinstance(method_name, str):
            display_name = method_name
        else:
            display_name = func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # For PyQt methods, handle the case where extra arguments are passed
                # by filtering to only the arguments the original function expects
                sig = inspect.signature(func)
                # Bind only the arguments that the function can accept
                bound_args = sig.bind_partial(*args[:len(sig.parameters)], **kwargs)
                bound_args.apply_defaults()
                return func(*bound_args.args, **bound_args.kwargs)
            except Exception as e:
                print(f"Error in {display_name}: {str(e)}")
                import traceback
                traceback.print_exc(file=sys.stderr)
                # Return None to prevent further errors
                return None
        return wrapper
    
    # Handle both @handle_exceptions and @handle_exceptions("name") syntax
    if callable(method_name):
        # Used without arguments: @handle_exceptions
        return decorator(method_name)
    else:
        # Used with arguments: @handle_exceptions("name") or just called to get decorator
        return decorator