import requests

ngrok_url = "https://e198-34-125-170-243.ngrok-free.app/"  # Replace with the URL printed by ngrok in Colab


def call_llm_api(prompt):
    headers = {'Content-Type': 'application/json'}
    data = {'prompt': prompt}
    response = requests.post(f"{ngrok_url}/generate", headers=headers, json=data)
    print(response)
    if response.status_code == 200:
        return response.json()['response']
    else:
        return f"Error: {response.status_code} - {response.text}"
