import sqlite3
import json
import numpy as np
import pandas as pd
import os

db_path = '/root/code/polyga_project_mmpolymer/results_formal_run/BRICSPolyPlanet/planetary_database.sqlite'

def main():
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return

    try:
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        
        rows = cur.execute("SELECT properties FROM polymer WHERE generation = 0").fetchall()
        
        dc_values = []
        for r in rows:
            if r['properties']:
                try:
                    props = json.loads(r['properties'])
                    if 'DC' in props and props['DC'] is not None:
                        dc_values.append(float(props['DC']))
                except:
                    pass
        con.close()
        
        if not dc_values:
            print("No DC values found.")
            return

        series = pd.Series(dc_values)
        percentiles = [75, 80, 85, 90]
        results = np.percentile(series, percentiles)
        
        print(f"Gen 0 DC Stats (Count: {len(series)})")
        for p, val in zip(percentiles, results):
            print(f"P{p}: {val:.4f}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
