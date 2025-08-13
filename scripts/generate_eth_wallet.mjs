#!/usr/bin/env node
import { randomBytes, Mnemonic, HDNodeWallet } from "ethers";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";

const DERIVATION_PATH = "m/44'/60'/0'/0/0";
const WALLETS_DIR = "/workspace/wallets/ethereum";

function formatUtcTimestamp() {
	const now = new Date();
	const yyyy = now.getUTCFullYear();
	const mm = String(now.getUTCMonth() + 1).padStart(2, "0");
	const dd = String(now.getUTCDate()).padStart(2, "0");
	const HH = String(now.getUTCHours()).padStart(2, "0");
	const MM = String(now.getUTCMinutes()).padStart(2, "0");
	const SS = String(now.getUTCSeconds()).padStart(2, "0");
	return `${yyyy}${mm}${dd}_${HH}${MM}${SS}`;
}

function main() {
	mkdirSync(WALLETS_DIR, { recursive: true });
	const entropy = randomBytes(32);
	const mnemonic = Mnemonic.fromEntropy(entropy);
	const wallet = HDNodeWallet.fromMnemonic(mnemonic, DERIVATION_PATH);

	const timestamp = formatUtcTimestamp();
	let seedFile = path.join(WALLETS_DIR, `seed_${timestamp}.txt`);
	try {
		writeFileSync(seedFile, mnemonic.phrase + "\n", { mode: 0o600, flag: "wx" });
	} catch (err) {
		if (err && err.code === "EEXIST") {
			seedFile = path.join(WALLETS_DIR, `seed_${timestamp}_${process.pid}.txt`);
			writeFileSync(seedFile, mnemonic.phrase + "\n", { mode: 0o600, flag: "wx" });
		} else {
			throw err;
		}
	}

	console.log(`Ethereum address: ${wallet.address}`);
	console.log(`Derivation path: ${DERIVATION_PATH}`);
	console.log(`Seed phrase saved to: ${seedFile}`);
}

main();