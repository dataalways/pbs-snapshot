#!/usr/bin/env python3

"""
Identifies the proposers of missed slots and tracks the path of those slots
through the MEV supply chain
"""

import argparse
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import time
from typing import Optional
import urllib3

import cryo
import pandas as pd
import requests

PROGNAME = "pbs_snapshot"
LOGGING_FORMAT = f"{PROGNAME}: %(levelname)s: %(message)s"

HTTP_CONNECT_TIMEOUT = 3.05  # in seconds
HTTP_READ_TIMEOUT = 27  # in seconds

EPOCH_SIZE = 32  # in slots
MERGE_SLOT = 4700013
SLOT0_TIME = 1606824023  # in seconds since the Unix epoch
SLOT_INTERVAL = 12  # in seconds

MERKLE_ETH_RPC_URL = "https://eth.merkle.io"

BEACONCHAIN_URL = "https://beaconcha.in"
BEACONCHAIN_RATE_LIMIT = 5  # in requests per second

PUBKEY_INDEXES_FILE = Path("proposer_indexes.parquet")

RELAYS = {
    "bloxroute_max_profit": "https://bloxroute.max-profit.blxrbdn.com",
    "agnostic": "https://agnostic-relay.net",
    "ultrasound": "https://relay.ultrasound.money",
    "flashbots": "https://boost-relay.flashbots.net",
    "aestus": "https://mainnet.aestus.live",
    "bloxroute_regulated": "https://bloxroute.regulated.blxrbdn.com",
    "titan": "http://titanrelay.xyz",
    "eden": "https://relay.edennetwork.io",
    "manifold": "https://mainnet-relay.securerpc.com",
}

EXPORT_DIR = Path("export")


def slot_from_timestamp(ts: int) -> int:
    """
    Calculates the slot number from a Unix time
    """
    if ts < SLOT0_TIME:
        raise ValueError(
            f"expected Unix time greater than or equal to the time of the first slot ({SLOT0_TIME})"
        )
    return (ts - SLOT0_TIME) // SLOT_INTERVAL


class RestApiClient:
    """
    Basic building block for implementing unauthenticated REST API clients
    """

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def _get(
        self, endpoint: str = "/", params: Optional[dict] = None
    ) -> requests.Response:
        return requests.get(
            self.base_url + endpoint,
            params=params,
            timeout=(HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT),
        )


@dataclass
class BidTraceV2:
    slot: int
    parent_hash: str
    block_hash: str
    builder_pubkey: str
    proposer_pubkey: str
    proposer_fee_recipient: str
    gas_limit: int
    gas_used: int
    value: str
    block_number: int
    num_tx: int


class MevBoostRelayDataClientV1(RestApiClient):
    """
    Client to download data from a HTTP server conforming to the MEV-Boost Relay API Spec v1
    """

    def __init__(self, base_url: str) -> None:
        super().__init__(base_url)
        self.endpoint_root = "/relay/v1/data"

    def proposer_payload_delivered(self, cursor: int) -> tuple[list[BidTraceV2], int]:
        """
        Get payloads delivered to proposers for a particular slot (`cursor`)
        """
        endpoint = self.endpoint_root + "/bidtraces/proposer_payload_delivered"
        params = {"cursor": cursor, "limit": 100}
        r = self._get(endpoint, params)
        if r.status_code != 200:
            return [], r.status_code
        return [
            BidTraceV2(
                slot=int(bt["slot"]),
                parent_hash=bt["parent_hash"],
                block_hash=bt["block_hash"],
                builder_pubkey=bt["builder_pubkey"],
                proposer_pubkey=bt["proposer_pubkey"],
                proposer_fee_recipient=bt["proposer_fee_recipient"],
                gas_limit=int(bt["gas_limit"]),
                gas_used=int(bt["gas_used"]),
                value=bt["value"],
                block_number=int(bt["block_number"]),
                num_tx=int(bt["num_tx"]),
            )
            for bt in r.json()
        ], r.status_code


def get_data_for_relay(
    relay: str, max_slot: int, lookback: int
) -> tuple[list[BidTraceV2], bool, bool]:
    """
    Gets payloads delivered to proposers via `relay` starting at slot `max_slot` and going back by `lookback` slots
    """
    client = MevBoostRelayDataClientV1(RELAYS[relay])
    cursor = max_slot
    min_slot = 0
    result = []
    relay_available = True
    rate_limited_maybe = False
    n_retries = 0
    while cursor > min_slot:
        bidtraces, status_code = client.proposer_payload_delivered(cursor)
        match (status_code):
            case 200:
                pass
            case 408 | 429 | 502:
                if n_retries > 2:
                    rate_limited_maybe = True
                    break
                n_retries += 1
                time.sleep(5)
                continue
            case 503:
                relay_available = False
                break
            case _:
                raise RuntimeError(f"unexpected HTTP status code: {status_code}")
        slots = [bt.slot for bt in bidtraces]
        if len(result) == 0:
            if len(slots) == 0:
                break
            min_slot = max(slots) - lookback
        result.extend(bidtraces)
        cursor = min(slots) - 1
    return result, relay_available, rate_limited_maybe


@dataclass
class BeaconChainApiSlotResponseV1:
    attestationscount: int
    attesterslashingscount: int
    blockroot: str
    depositscount: int
    epoch: int
    eth1data_blockhash: str
    eth1data_depositcount: int
    eth1data_depositroot: str
    exec_base_fee_per_gas: int
    exec_block_hash: str
    exec_block_number: int
    exec_extra_data: str
    exec_fee_recipient: str
    exec_gas_limit: int
    exec_gas_used: int
    exec_logs_bloom: str
    exec_parent_hash: str
    exec_random: str
    exec_receipts_root: str
    exec_state_root: str
    exec_timestamp: int
    exec_transactions_count: int
    graffiti: str
    graffiti_text: str
    parentroot: str
    proposer: int
    proposerslashingscount: int
    randaoreveal: str
    signature: str
    slot: int
    stateroot: str
    status: str
    syncaggregate_bits: str
    syncaggregate_participation: float
    syncaggregate_signature: str
    voluntaryexitscount: int
    withdrawalcount: int


class BeaconChainEpochClientV1(RestApiClient):
    """
    Client to download consensus-layer information about epochs from beaconcha.in's REST API v1
    """

    def __init__(self) -> None:
        super().__init__(BEACONCHAIN_URL)
        self.endpoint_root = "/api/v1/epoch"

    def slots_in_epoch(
        self, epoch: int
    ) -> tuple[list[BeaconChainApiSlotResponseV1], int]:
        """
        Get latest or finalized blocks in an epoch
        """
        endpoint = f"{self.endpoint_root}/{epoch}/slots"
        r = self._get(endpoint)
        if r.status_code != 200:
            return [], r.status_code
        return [
            BeaconChainApiSlotResponseV1(
                attestationscount=int(s["attestationscount"]),
                attesterslashingscount=int(s["attesterslashingscount"]),
                blockroot=s["blockroot"],
                depositscount=int(s["depositscount"]),
                epoch=int(s["epoch"]),
                eth1data_blockhash=s["eth1data_blockhash"],
                eth1data_depositcount=int(s["eth1data_depositcount"]),
                eth1data_depositroot=s["eth1data_depositroot"],
                exec_base_fee_per_gas=int(s["exec_base_fee_per_gas"]),
                exec_block_hash=s["exec_block_hash"],
                exec_block_number=int(s["exec_block_number"]),
                exec_extra_data=s["exec_extra_data"],
                exec_fee_recipient=s["exec_fee_recipient"],
                exec_gas_limit=int(s["exec_gas_limit"]),
                exec_gas_used=int(s["exec_gas_used"]),
                exec_logs_bloom=s["exec_logs_bloom"],
                exec_parent_hash=s["exec_parent_hash"],
                exec_random=s["exec_random"],
                exec_receipts_root=s["exec_receipts_root"],
                exec_state_root=s["exec_state_root"],
                exec_timestamp=int(s["exec_timestamp"]),
                exec_transactions_count=int(s["exec_transactions_count"]),
                graffiti=s["graffiti"],
                graffiti_text=s["graffiti_text"],
                parentroot=s["parentroot"],
                proposer=int(s["proposer"]),
                proposerslashingscount=int(s["proposerslashingscount"]),
                randaoreveal=s["randaoreveal"],
                signature=s["signature"],
                slot=int(s["slot"]),
                stateroot=s["stateroot"],
                status=s["status"],
                syncaggregate_bits=s["syncaggregate_bits"],
                syncaggregate_participation=float(s["syncaggregate_participation"]),
                syncaggregate_signature=s["syncaggregate_signature"],
                voluntaryexitscount=int(s["voluntaryexitscount"]),
                withdrawalcount=int(s["withdrawalcount"]),
            )
            for s in r.json()["data"]
        ], r.status_code


class Report:
    """
    Human-readable summary of analysis
    """

    def __init__(self) -> None:
        self.unavailable_relays = []
        self.min_block_number = None
        self.max_block_number = None
        self.n_payloads_delivered_by_relay = {}
        self.n_payloads_delivered_by_relay_bloxroute_dedup = {}
        self.mev_block_fraction = None
        self.missed_slot_fraction = None
        self.n_missed_mevboost_payloads_delivered_by_relay = {}
        self.n_proposers_with_missed_slots = None
        self.n_untagged_proposers_with_missed_slots = None
        self.n_missed_slots_by_entity = {}

    def __str__(self) -> str:
        result = []

        if self.unavailable_relays:
            result.append(
                "The following relays were unavailable: "
                + ", ".join(self.unavailable_relays)
            )
        else:
            result.append("All queried relays were available!")

        if self.min_block_number and self.max_block_number:
            result.append(
                "Range of blocks analyzed: "
                + f"from {self.min_block_number} to {self.max_block_number}"
            )

        if self.n_payloads_delivered_by_relay:
            result.append(
                "Number of payloads delivered by relay: "
                + ", ".join(
                    f"{k}: {v}" for k, v in self.n_payloads_delivered_by_relay.items()
                )
            )

        if self.n_payloads_delivered_by_relay_bloxroute_dedup:
            result.append(
                "Number of payloads delivered by only one relay: "
                + ", ".join(
                    f"{k}: {v}"
                    for k, v in self.n_payloads_delivered_by_relay_bloxroute_dedup.items()
                )
            )

        if self.mev_block_fraction is not None:
            result.append(
                "Share of MEV-Boost blocks: " + f"{self.mev_block_fraction:.2%}"
            )

        if self.missed_slot_fraction is not None:
            result.append(
                "Share of missed slots: " + f"{self.missed_slot_fraction:.2%}"
            )

        if self.n_missed_mevboost_payloads_delivered_by_relay:
            result.append(
                "Number of missed MEV-Boost payloads delivered by relay: "
                + ", ".join(
                    f"{k}: {v}"
                    for k, v in self.n_missed_mevboost_payloads_delivered_by_relay.items()
                )
            )

        if self.n_proposers_with_missed_slots is not None:
            result.append(
                "Number of proposers that missed their slots: "
                + f"{self.n_proposers_with_missed_slots}"
            )

        if self.n_untagged_proposers_with_missed_slots is not None:
            result.append(
                "Number of untagged proposers that missed their slots: "
                + f"{self.n_untagged_proposers_with_missed_slots}"
            )

        if self.n_missed_slots_by_entity:
            result.append(
                "Number of missed slots by entity: "
                + ", ".join(
                    f"{k}: {v}" for k, v in self.n_missed_slots_by_entity.items()
                )
            )

        return "\n".join(result)


def cryo_collect_blocks(
    blocks: list[str],
    rpc: str,
    requests_per_second: int,
    exclude_columns: Optional[list[str]] = None,
    logger: logging.Logger = logging.getLogger(__name__),
) -> pd.DataFrame:
    """
    Collects data about one or more blocks using cryo
    """

    result = pd.DataFrame()
    try:
        result = cryo.collect(
            "blocks",
            blocks=blocks,
            rpc=rpc,
            output_format="pandas",
            hex=True,
            requests_per_second=requests_per_second,
            exclude_columns=exclude_columns,
        )

    # cryo may panic while downloading data. We can't import the corresponding
    # exception type so we make a best effort using introspection.
    except Exception as e:
        if e.__class__.__name__ == "RustPanic":
            # Assuming correct inputs, cryo likely panicked because it received
            # a JSON-RPC error related to rate limiting. We can't prove this
            # because the exception doesn't carry information about the reason
            # for the panic (the panicking thread writes that information to
            # stderr). In any event, we retry once with merkle if the given RPC
            # endpoint was not already merkle.
            logger.warning(
                "thread panicked possibly as a result of JSON-RPC rate limiting"
            )
            if not rpc.startswith(MERKLE_ETH_RPC_URL):
                logger.info("retrying download with eth.merkle.io")
                result = cryo.collect(
                    "blocks",
                    blocks=blocks,
                    rpc=MERKLE_ETH_RPC_URL,
                    output_format="pandas",
                    hex=True,
                    requests_per_second=2,
                    exclude_columns=exclude_columns,
                )
            else:
                raise e
        else:
            raise e

    return result


def ensure_dir(d: Path) -> None:
    """Ensures that a directory exists"""
    if not d.exists():
        os.mkdir(d)


def export_csv(
    df: pd.DataFrame, name: str, logger: logging.Logger = logging.getLogger(__name__)
) -> None:
    """Export a pandas DataFrame to disk as a CSV file"""
    ensure_dir(EXPORT_DIR)
    path_or_buf = f"{EXPORT_DIR}/{name}.csv"
    logger.info("writing data to '%s'", path_or_buf)
    df.to_csv(path_or_buf, index=False)


def main(
    report: Report,
    rpc: str,
    requests_per_second: int,
    lookback: int,
    max_slot: Optional[int] = None,
    export_data: bool = True,
    logger: logging.Logger = logging.getLogger(__name__),
) -> None:
    latest_block = cryo_collect_blocks(
        blocks=["latest:"],
        rpc=rpc,
        requests_per_second=requests_per_second,
        exclude_columns=["author", "gas_used", "base_fee_per_gas", "chain_id"],
    ).iloc[0]
    latest_filled_slot = slot_from_timestamp(latest_block["timestamp"]) - 1
    if max_slot:
        if max_slot > latest_filled_slot:
            raise ValueError(
                f"given max slot ({max_slot}) is after the latest filled slot ({latest_filled_slot})"
            )
    else:
        max_slot = latest_filled_slot

    df = pd.DataFrame()
    dfs = []

    logger.info("downloading payload delivery data from MEV-Boost relays")
    for relay in RELAYS:
        data_for_relay, relay_available, rate_limited_maybe = get_data_for_relay(
            relay, max_slot=max_slot, lookback=lookback
        )
        if not relay_available:
            report.unavailable_relays.append(relay)
            logger.warning(
                "relay '%s' is unavailable or went down during connection", relay
            )
        if rate_limited_maybe:
            logger.warning(
                "cancelled download from relay '%s' due to possible rate limiting or a lossy connection",
                relay,
            )
        if not data_for_relay:
            logger.info(
                "relay '%s' returned no data; MEV share may be an underestimate", relay
            )
            continue
        df = pd.DataFrame(data_for_relay)
        df["relay"] = relay
        dfs.append(df)

    df_slots = pd.concat(dfs)
    df_slots = df_slots[df_slots["slot"] > df_slots["slot"].max() - lookback]
    df_slots = df_slots.sort_values(by="slot")
    report.n_payloads_delivered_by_relay = dict(df_slots["relay"].value_counts())

    # Treat bloXroute Max Profit and bloXroute Regulated relays as one relay
    df_slots.loc[
        df_slots["relay"].isin(["bloxroute_max_profit", "bloxroute_regulated"]), "relay"
    ] = "bloxroute"
    df_slots = df_slots.drop_duplicates(subset=["slot", "relay"], keep="first")
    df_slots_bloxroute_dedup = df_slots.drop_duplicates(subset=["slot"], keep=False)
    report.n_payloads_delivered_by_relay_bloxroute_dedup = dict(
        df_slots_bloxroute_dedup["relay"].value_counts()
    )

    logger.info("downloading block data from RPC endpoint")
    min_block_no, max_block_no = (
        df_slots["block_number"].min(),
        df_slots["block_number"].max(),
    )
    df_blocks = cryo_collect_blocks(
        blocks=[f"{min_block_no}:{max_block_no + 1}"],
        rpc=rpc,
        requests_per_second=requests_per_second,
    )

    if export_data:
        df_blocks_enriched = df_blocks.join(
            df_slots.set_index("block_hash"),
            on="block_hash",
            how="left",
            rsuffix="_slots",
        )
        df_blocks_enriched = df_blocks_enriched.drop(
            columns=["gas_used_slots", "block_number_slots"]
        )
        try:
            min_slot_no = int(df_blocks_enriched["slot"].min())
            max_slot_no = int(df_blocks_enriched["slot"].max())
            export_csv(
                df_blocks_enriched, f"slots_{min_slot_no}-{max_slot_no}_snapshot"
            )
        except Exception as e:
            logger.warning("unable to export enriched block data")
            logger.warning("reason: %s: %s", e.__class__.__name__, e)
            logger.info("continuing analysis")

    report.min_block_number = min_block_no
    report.max_block_number = max_block_no

    # Compute fraction of MEV-Boost blocks
    mev_blocks = df_slots.drop_duplicates(subset="block_number", keep="first")
    mev_block_fraction = len(mev_blocks) / len(df_blocks)
    report.mev_block_fraction = mev_block_fraction

    # Compute fraction of missed slots
    df_blocks["block_time"] = df_blocks["timestamp"].diff(1)
    missed_slots = (
        sum(df_blocks["block_time"].dropna() - SLOT_INTERVAL) // SLOT_INTERVAL
    )
    missed_slot_fraction = missed_slots / lookback
    report.missed_slot_fraction = missed_slot_fraction

    # Count payloads delivered to each proposer (possibly by more than one relay) but not included on the chain
    block_hashes = set(df_blocks["block_hash"])
    df_missed_payloads = df_slots[~df_slots["block_hash"].isin(block_hashes)]

    if export_data and not df_missed_payloads.empty:
        min_slot_no_missed = int(df_missed_payloads["slot"].min())
        max_slot_no_missed = int(df_missed_payloads["slot"].max())
        try:
            export_csv(
                df_missed_payloads,
                f"slots_{min_slot_no_missed}-{max_slot_no_missed}_missed",
            )
        except Exception as e:
            logger.warning("unable to export missed slot data")
            logger.warning("reason: %s: %s", e.__class__.__name__, e)
            logger.info("continuing analysis")

    report.n_missed_mevboost_payloads_delivered_by_relay = dict(
        df_missed_payloads["relay"].value_counts()
    )

    ini_ts, fin_ts = df_blocks.iloc[0]["timestamp"], df_blocks.iloc[-1]["timestamp"]
    expected_timestamps = range(ini_ts, fin_ts, SLOT_INTERVAL)
    missing_timestamps = [
        ts for ts in expected_timestamps if ts not in set(df_blocks["timestamp"])
    ]
    missing_slots = [slot_from_timestamp(ts) for ts in missing_timestamps]
    epochs = [s // EPOCH_SIZE for s in missing_slots]

    validator_indexes_for_missed_payloads = set()
    beaconchain_client = BeaconChainEpochClientV1()
    logger.info("downloading missed slot data from beaconcha.in")
    for epoch in epochs:
        epoch_data, status_code = beaconchain_client.slots_in_epoch(epoch)
        match (status_code):
            case 200:
                pass
            case _:
                raise RuntimeError(f"unexpected HTTP status code: {status_code}")
        df_epoch = pd.DataFrame(epoch_data)
        validator_indexes_for_missed_payloads.update(
            set(df_epoch[df_epoch["status"] != "1"]["proposer"])
        )
        time.sleep(1 / BEACONCHAIN_RATE_LIMIT)

    n_proposers_with_missed_slots = len(validator_indexes_for_missed_payloads)
    report.n_proposers_with_missed_slots = n_proposers_with_missed_slots

    validator_indexes = pd.read_parquet(
        PUBKEY_INDEXES_FILE, columns=["entity", "validator_index"]
    )
    tagged_proposers = validator_indexes[
        validator_indexes["validator_index"].isin(validator_indexes_for_missed_payloads)
    ]
    n_untagged_proposers_with_missed_slots = n_proposers_with_missed_slots - len(
        tagged_proposers
    )
    report.n_untagged_proposers_with_missed_slots = (
        n_untagged_proposers_with_missed_slots
    )
    report.n_missed_slots_by_entity = dict(tagged_proposers["entity"].value_counts())


def validate_rpc_url(url: urllib3.util.url.Url) -> None:
    """
    Ensure that a URL to an RPC service is either a complete HTTP, HTTPS or WS
    URL or an absolute local Unix path to a Unix IPC pipe
    """
    match (url.scheme):
        case "http" | "https" | "ws":
            if not url.host:
                raise ValueError("missing host for RPC URL with HTTP/HTTPS/WS scheme")
        case "file":
            if not url.path:
                raise ValueError(
                    "missing absolute path to Unix IPC pipe for RPC URL with file scheme"
                )
            if not url.path.endswith(".ipc"):
                raise ValueError(
                    "RPC URL with file scheme does not refer to a Unix IPC pipe"
                )
        case None:
            if url.host:
                raise ValueError("missing scheme for RPC URL with host")
            if not url.path:
                raise ValueError("incomplete RPC URL")
            if not url.path.endswith(".ipc"):
                raise ValueError(
                    "RPC URL with no scheme or host must be an absolute path to a Unix IPC pipe"
                )
        case _:
            raise ValueError("unsupported scheme in RPC URL")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=Path(__file__).name,
        description="Collect data for making assertions about the health of the MEV-Boost relay network",
    )
    parser.add_argument(
        "-E",
        "--no-export-data",
        action="store_true",
        help="Do not export CSV data (defaults to false)",
    )
    parser.add_argument(
        "-l",
        "--lookback",
        default=50,
        type=int,
        help="Number of slots by which to look back (defaults to 50)",
    )
    parser.add_argument(
        "-m",
        "--max-slot",
        type=int,
        help="Largest slot number to consider for which a block has been added to the chain (defaults to the latest)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Do not log normal events",
    )
    parser.add_argument(
        "-r",
        "--rpc",
        default=os.getenv("ETH_RPC_URL") or MERKLE_ETH_RPC_URL,
        help="RPC endpoint from which to fetch block metadata (defaults to the value of the ETH_RPC_URL environment variable or to eth.merkle.io)",
    )
    parser.add_argument(
        "-R",
        "--requests-per-second",
        default=2,
        type=int,
        help="Request rate for the given RPC endpoint (defaults to 2)",
    )
    args = parser.parse_args()

    if not args.lookback > 0:
        raise ValueError("lookback must be a positive integer")

    if args.max_slot:
        if not args.max_slot - args.lookback > MERGE_SLOT:
            raise ValueError(f"cannot look beyond merge slot ({MERGE_SLOT})")

    if not args.requests_per_second > 0:
        raise ValueError("program must make at least one request per second")

    rpc_parsed = urllib3.util.parse_url(args.rpc)
    validate_rpc_url(rpc_parsed)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO, format=LOGGING_FORMAT
    )

    report = Report()

    try:
        main(
            report,
            args.rpc,
            args.requests_per_second,
            args.lookback,
            args.max_slot,
            not args.no_export_data,
        )
    finally:
        print(report)
