"""Point your Twilio number at the phone line (or print how to do it by hand).

    python -m callie.phone.setup --url https://<name>.trycloudflare.com

Sets the number's "A call comes in" webhook to <url>/twilio/voice (POST) and its status
callback to <url>/twilio/status. Needs TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN and
TWILIO_PHONE_NUMBER in .env; without them it prints the console steps instead.
"""

from __future__ import annotations

import argparse
import sys

from callie.config import get_callie_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True, help="the public https URL of the phone gateway (the tunnel)")
    args = parser.parse_args()
    url = args.url.rstrip("/")
    settings = get_callie_settings()
    voice, status = f"{url}/twilio/voice", f"{url}/twilio/status"
    if not (settings.twilio_account_sid and settings.twilio_auth_token and settings.twilio_phone_number):
        print("TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN / TWILIO_PHONE_NUMBER aren't all set, so set it by hand:\n"
              "  Twilio console > Phone Numbers > Manage > Active numbers > your number > Voice configuration:\n"
              f"    A call comes in:  Webhook  {voice}  (HTTP POST)\n"
              f"    Call status changes:       {status}  (HTTP POST)")
        sys.exit(0)
    from twilio.rest import Client

    client = Client(settings.twilio_account_sid, settings.twilio_auth_token.get_secret_value())
    numbers = client.incoming_phone_numbers.list(phone_number=settings.twilio_phone_number, limit=1)
    if not numbers:
        raise SystemExit(f"{settings.twilio_phone_number} isn't a number on this Twilio account")
    numbers[0].update(voice_url=voice, voice_method="POST", status_callback=status, status_callback_method="POST")
    print(f"{settings.twilio_phone_number} now calls {voice}")


if __name__ == "__main__":
    main()
