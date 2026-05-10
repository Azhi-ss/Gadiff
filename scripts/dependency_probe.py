import sys
import os
from unittest.mock import patch

def run_probe():
    print("--- Running Dynamic Dependency Probe ---")
    before = set(sys.modules.keys())

    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=1):
        
        # 针对 polyga / MMPolymer 进行模拟引入
        try:
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/models')))
            
            # import everything to trigger dependencies
            import src.optimization.run_optimization as run_opt
            from src.models import MMPolymer
            from src.models import polyga
        except Exception as e:
            print("Import encountered some issues, but continuing:", e)

    after = set(sys.modules.keys())
    new_deps = after - before
    
    # 提取顶层包
    top_level = set()
    for m in new_deps:
        base = m.split('.')[0]
        if not base.startswith('_'):
            top_level.add(base)
            
    # 去除标准库
    import sysconfig
    stdlib = sysconfig.get_paths()['stdlib']
    
    external_deps = []
    for pkg in sorted(top_level):
        if pkg in sys.builtin_module_names:
            continue
        try:
            mod = __import__(pkg)
            if hasattr(mod, '__file__') and mod.__file__:
                if 'site-packages' in mod.__file__ or 'dist-packages' in mod.__file__:
                    external_deps.append(pkg)
        except Exception:
            pass
            
    print("\n[Dynamic Probe] External packages loaded:")
    for dep in external_deps:
        print(f" - {dep}")

if __name__ == "__main__":
    run_probe()
