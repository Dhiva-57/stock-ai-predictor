import yfinance as yf
import pandas as pd
import os

print("Starting...", flush=True)

try:
    nvda = yf.Ticker("NVDA")

    print("Getting option expiration dates...", flush=True)

    expiries = nvda.options

    if not expiries:
        print("No option data found.")
        exit()

    nearest_expiry = expiries[0]

    print(f"Nearest Expiry: {nearest_expiry}", flush=True)

    chain = nvda.option_chain(nearest_expiry)

    calls = pd.DataFrame(chain.calls)
    calls["Type"] = "Call"

    puts = pd.DataFrame(chain.puts)
    puts["Type"] = "Put"

    full_chain_file = pd.concat(
        [calls, puts],
        ignore_index=True
    )

    csv_file = "NVDA_Live_Options_Chain.csv"

    full_chain_file.to_csv(
        csv_file,
        index=False
    )

    print(
        f"Rows Saved: {len(full_chain_file)}",
        flush=True
    )

    print(
        "CSV Location:",
        os.path.abspath(csv_file),
        flush=True
    )

    print(
        "Success! File created.",
        flush=True
    )

except Exception as e:
    print(
        f"Error: {e}",
        flush=True
    )