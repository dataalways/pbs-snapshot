# PBS Snapshot

PBS Snapshot is a command-line program written in [Python](https://www.python.org/) that delivers a simplistic snapshot of proposer/builder separation on [Ethereum](https://ethereum.org/). It runs out of the box with no authentication requirements, however for larger historic snapshots we recommend setting a custom JSON-RPC endpoint and rate limit.

The snapshot is created by:

- Querying all relays to get recently delivered MEV-Boost payloads
- Leveraging [cryo](https://github.com/paradigmxyz/cryo) to download block data that we cross-reference against the relay payload data
- Calculating the share of blocks proposed using MEV-Boost and counting blocks delivered by each relay to watch for abnormalities
- Downloading missed slot data from the [beaconcha.in API](https://beaconcha.in/api/v1/docs/index.html) to retrieve proposer indexes for missed slots
- Identifying tagged proposers who missed their slots to isolate proposer sets who may be experiencing outages (sourced from [Dune Analytics](https://dune.com/queries/3691192) and curated by [Hildobby](https://x.com/hildobby_))

## Installation

PBS Snapshot requires Git and Python 3.12.

### Clone and enter the repository

```console
$ git clone --depth 1 https://github.com/dataalways/pbs-snapshot
$ cd pbs-snapshot
```

### Install the project's dependencies

We strongly recommend installing the dependencies to a virtual environment.

For users of [pipenv](https://pipenv.pypa.io), we provide both *Pipfile* and *Pipfile.lock*. To guarantee reproducible execution, install the dependencies like so:

```console
$ pipenv install --ignore-pipfile
```

We also provide a minimal *pyproject.toml* file for users of modern project managers such as [Rye](https://rye.astral.sh), [Hatch](https://hatch.pypa.io) and [PDM](https://pdm-project.org).

## Usage

In the following examples, each command is run in a virtual environment containing the project's dependencies.

### Example 1: Live picture of the blockchain

```console
(pbs-snapshot) $ ./pbs_snapshot.py
```

```
pbs_snapshot: INFO: downloading payload delivery data from MEV-Boost relays
pbs_snapshot: INFO: downloading block data from RPC endpoint
pbs_snapshot: INFO: writing data to 'export/slots_9275070-9275119_snapshot.csv'
pbs_snapshot: INFO: downloading missed slot data from beaconcha.in
All queried relays were available!
Range of blocks analyzed: from 20069909 to 20069958
Number of payloads delivered by relay: ultrasound: 30, bloxroute_max_profit: 26, bloxroute_regulated: 17, agnostic: 17, flashbots: 10, aestus: 6
Number of payloads delivered by only one relay: bloxroute: 10, ultrasound: 10, flashbots: 3
Share of MEV-Boost blocks: 96.00%
Share of missed slots: 0.00%
Number of proposers that missed their slots: 0
Number of untagged proposers that missed their slots: 0
```

Analysis of output:

- There were 0 missed slots in the 10 minutes prior to program execution
- 96% of blocks were built using MEV-Boost, which is in line with expectations
- Network health was nominal

### Example 2: [bloXroute BDN issues](https://gist.github.com/benhenryhunter/687299bcfe064674537dc9348d771e83) on March 27, 2024

```console
(pbs-snapshot) $ ./pbs_snapshot.py --max-slot 8729000 --lookback 1000
```

```
pbs_snapshot: INFO: downloading payload delivery data from MEV-Boost relays
pbs_snapshot: INFO: downloading block data from RPC endpoint
pbs_snapshot: INFO: writing data to 'export/slots_8728001-8729000_snapshot.csv'
pbs_snapshot: INFO: writing data to 'export/slots_8728135-8728990_missed.csv'
pbs_snapshot: INFO: downloading missed slot data from beaconcha.in
All queried relays were available!
Range of blocks analyzed: from 19527062 to 19527984
Number of payloads delivered by relay: ultrasound: 564, agnostic: 330, bloxroute_max_profit: 275, bloxroute_regulated: 211, flashbots: 189, aestus: 97, titan: 8, manifold: 3, eden: 2
Number of payloads delivered by only one relay: ultrasound: 200, bloxroute: 105, flashbots: 65, agnostic: 13, eden: 2, aestus: 1, manifold: 1, titan: 1
Share of MEV-Boost blocks: 85.81%
Share of missed slots: 7.70%
Number of missed MEV-Boost payloads delivered by relay: bloxroute: 59, ultrasound: 18, flashbots: 10, agnostic: 9, aestus: 7
Number of proposers that missed their slots: 77
Number of untagged proposers that missed their slots: 16
Number of missed slots by entity: Lido: 22, Figment: 11, Coinbase: 9, Binance: 6, Kiln: 5, CoinSpot: 2, ether.fi: 2, Rocket Pool: 1, Kraken: 1, OKX: 1, Staked.us: 1
```

Analysis of output:

- Data was available for all relays
- There were 77 missed slots; bloXroute relayed the payloads for 59 (77%) of those slots
- The proposers were a diverse set with no group significantly over-represented versus their market share
- The most likely cause of the network instability was an issue with bloXroute's relays

### Example 3: [Coinbase Outage](https://ethstaker.notion.site/Portion-of-the-network-is-offline-e08da6aab1124097888b3bdd2a3febf7) on March 8, 2024

```console
(pbs-snapshot) $ ./pbs_snapshot.py --max-slot 8593000 --lookback 500
```

```
pbs_snapshot: INFO: downloading payload delivery data from MEV-Boost relays
pbs_snapshot: INFO: relay 'bloxroute_max_profit' returned no data; MEV share may be an underestimate
pbs_snapshot: INFO: relay 'bloxroute_regulated' returned no data; MEV share may be an underestimate
pbs_snapshot: INFO: downloading block data from RPC endpoint
pbs_snapshot: INFO: writing data to 'export/slots_8592501-8593000_snapshot.csv'
pbs_snapshot: INFO: downloading missed slot data from beaconcha.in
All queried relays were available!
Range of blocks analyzed: from 19393170 to 19393629
Number of payloads delivered by relay: ultrasound: 213, flashbots: 120, agnostic: 110, aestus: 24, titan: 1, eden: 1, manifold: 1
Number of payloads delivered by only one relay: ultrasound: 94, flashbots: 86, agnostic: 4, aestus: 3, titan: 1, eden: 1
Share of MEV-Boost blocks: 66.96%
Share of missed slots: 8.00%
Number of proposers that missed their slots: 41
Number of untagged proposers that missed their slots: 5
Number of missed slots by entity: Coinbase: 34, Bitcoin Suisse: 1, Revolut: 1
```

Analysis of output:

- The requested slots are outside of bloXroute's supported data range, so they are missing from the dataset
- The share of MEV-Boost blocks is underestimated because it is missing blocks only relayed by bloXroute
- The missed slots data is accurate because we calculate the value through the JSON-RPC data
- We see that 34 of 41 missed slots had Coinbase as the expected proposer, implying that they were the source of the outage

### Help text

```console
(pbs-snapshot) $ ./pbs_snapshot.py -h
```

```
usage: pbs_snapshot.py [-h] [-E] [-l LOOKBACK] [-m MAX_SLOT] [-q] [-r RPC] [-R REQUESTS_PER_SECOND]

Collect data for making assertions about the health of the MEV-Boost relay network

options:
  -h, --help            show this help message and exit
  -E, --no-export-data  Do not export CSV data (defaults to false)
  -l LOOKBACK, --lookback LOOKBACK
                        Number of slots by which to look back (defaults to 50)
  -m MAX_SLOT, --max-slot MAX_SLOT
                        Largest slot number to consider for which a block has been added to the chain (defaults to the latest)
  -q, --quiet           Do not log normal events
  -r RPC, --rpc RPC     RPC endpoint from which to fetch block metadata (defaults to the value of the ETH_RPC_URL environment variable
                        or to eth.merkle.io)
  -R REQUESTS_PER_SECOND, --requests-per-second REQUESTS_PER_SECOND
                        Request rate for the given RPC endpoint (defaults to 2)
```

## License

PBS Snapshot is provided under the [MIT License](./LICENSE.txt).

The included proposer indexes are derived from the Dune Analytics spellbook under the [additional use grant](https://github.com/duneanalytics/spellbook/blob/main/LICENSE).

## Acknowledgements

We acknowledge significant source code and documentation contributions from a frequent but anonymous Data Always collaborator.
