#!/usr/bin/env python3
import os
import sys
import time
from pathlib import Path
from typing import Tuple

try:
	from bip_utils import (
		Bip39MnemonicGenerator,
		Bip39SeedGenerator,
		Bip39WordsNum,
		Bip44,
		Bip44Coins,
		Bip44Changes,
	)
except ImportError:
	sys.stderr.write("Required package 'bip-utils' is not installed. Install with: python3 -m pip install --user bip-utils\n")
	sys.exit(1)

DERIVATION_PATH = "m/44'/60'/0'/0/0"
WALLETS_DIR = Path("/workspace/wallets/ethereum")

def generate_mnemonic_and_address() -> Tuple[str, str]:
	mnemonic = Bip39MnemonicGenerator().FromWordsNumber(Bip39WordsNum.WORDS_NUM_24)
	seed_bytes = Bip39SeedGenerator(mnemonic).Generate()
	bip44_ctx = Bip44.FromSeed(seed_bytes, Bip44Coins.ETHEREUM)
	addr_ctx = bip44_ctx.Purpose().Coin().Account(0).Change(Bip44Changes.CHAIN_EXT).AddressIndex(0)
	address = addr_ctx.PublicKey().ToAddress()
	return mnemonic, address

def write_secret_file_securely(file_path: Path, secret_text: str) -> None:
	flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
	fd = os.open(str(file_path), flags, 0o600)
	with os.fdopen(fd, "w", encoding="utf-8") as f:
		f.write(secret_text.strip() + "\n")

def main() -> int:
	WALLETS_DIR.mkdir(parents=True, exist_ok=True)
	mnemonic, address = generate_mnemonic_and_address()
	timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
	mnemonic_file = WALLETS_DIR / f"seed_{timestamp}.txt"
	try:
		write_secret_file_securely(mnemonic_file, mnemonic)
	except FileExistsError:
		mnemonic_file = WALLETS_DIR / f"seed_{timestamp}_{os.getpid()}.txt"
		write_secret_file_securely(mnemonic_file, mnemonic)
	print(f"Ethereum address: {address}")
	print(f"Derivation path: {DERIVATION_PATH}")
	print(f"Seed phrase saved to: {mnemonic_file}")
	return 0

if __name__ == "__main__":
	raise SystemExit(main())