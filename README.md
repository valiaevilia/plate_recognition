## Запуск клиента (client)

Эти шаги выполняются **на машине пользователя** — чтобы распознать номера в видео и получить CSV.

**Что нужно знать заранее**
- IP сервера, где запущен прокси (и порт), например `127.0.0.1:8080` или `192.168.0.50:8080`.
- Токен доступа к прокси: `PROXY_CLIENT_TOKEN`.
- Путь к видеофайлу (MP4 и др.).

> По умолчанию клиент шлёт запросы на `PROXY_URL = "http://127.0.0.1:8080/v1/chat/completions"`.
> Если прокси на другом ПК — в шаге **5** задайте переменную окружения `PROXY_URL`.

---

1 Открыть PowerShell в папке `client`
```powershell
cd C:\path\to\repo\client

2 Создать и активировать виртуальное окружение
python -m venv .venv
.\.venv\Scripts\Activate.ps1

3 Установить зависимости
python -m pip install --upgrade pip
pip install -r requirements.txt

client/requirements.txt должен содержать:
opencv-python
requests
pillow
python-dotenv

4 Положить видео

Скопируйте файл в client/videos/, например:

client/videos/test.mp4

5 (опционально) Указать адрес прокси и токен

Если прокси работает на другом ПК, задайте окружение перед запуском:

$env:PROXY_URL = "http://IP_СЕРВЕРА:8080/v1/chat/completions"
$env:PROXY_CLIENT_TOKEN = "demo-token-rotate-me"

6 Запустить распознавание
python .\plate_model.py --video ".\videos\test.mp4" --out ".\out.csv"