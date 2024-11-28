from xrpl.clients import JsonRpcClient
from xrpl.models.requests import AccountTx

# XRP Ledger RPC 서버 설정
client = JsonRpcClient("https://s1.ripple.com:51234/")

# XRP 지갑 주소
xrp_address = "rGN9R3ToTpZ8BuJCALLrhjRMjKUDxppmB1"

# 거래 내역 요청
request = AccountTx(account=xrp_address, ledger_index_min=-1, ledger_index_max=-1)
response = client.request(request)

# 거래 내역 출력
transactions = response.result["transactions"]
for tx in transactions:
    tx_data = tx['tx_json']
    memos = tx_data.get('Memos', [])
    if memos:
        for memo in memos:
            memo_data = memo.get('Memo', {})
            memo_text = memo_data.get('MemoData', '')
            print(f"Transaction Hash: {tx_data['hash']}")
            print(f"Memo: {memo_text}")
            print("-" * 40)
    else:
        print(f"Transaction Hash: {tx['hash']}")
        print("No Memo")
        print("-" * 40)
