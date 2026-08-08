import json
import urllib.parse
import urllib.request


class LiveData:

    def _get_json(self, url):

        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "PersonalAI-v2/1.0"
            }
        )

        with urllib.request.urlopen(
            request,
            timeout=10
        ) as response:

            return json.loads(
                response.read().decode("utf-8")
            )

    # =========================================================
    # GOLD PRICE
    # =========================================================

    def gold_price(self):

        try:

            # Gold API:
            # XAU = Gold
            data = self._get_json(
                "https://api.gold-api.com/price/XAU"
            )

            price_usd = (
                data.get("price")
            )

            if price_usd is None:

                return None

            price_usd = float(
                price_usd
            )

            # Get current USD -> INR.
            exchange = self._get_json(
                "https://api.frankfurter.app/latest"
                "?from=USD&to=INR"
            )

            usd_inr = float(
                exchange["rates"]["INR"]
            )

            # Troy ounce -> gram.
            grams_per_ounce = 31.1034768

            inr_per_gram_24k = (
                price_usd
                * usd_inr
                / grams_per_ounce
            )

            # Approximate purity conversion.
            inr_per_gram_22k = (
                inr_per_gram_24k
                * 22
                / 24
            )

            return (
                "Live gold spot price:\n"
                f"24K: ₹{inr_per_gram_24k:,.2f} per gram\n"
                f"22K: ₹{inr_per_gram_22k:,.2f} per gram\n"
                f"Spot price: ${price_usd:,.2f} per troy ounce\n"
                f"USD/INR: ₹{usd_inr:,.2f}\n\n"
                "Note: Indian jewellery prices can differ "
                "because of taxes, making charges, location, "
                "and dealer margins."
            )

        except Exception as error:

            print(
                "[LiveData] Gold API error:",
                error
            )

            return None

    # =========================================================
    # SILVER PRICE
    # =========================================================

    def silver_price(self):

        try:

            data = self._get_json(
                "https://api.gold-api.com/price/XAG"
            )

            price_usd = float(
                data["price"]
            )

            exchange = self._get_json(
                "https://api.frankfurter.app/latest"
                "?from=USD&to=INR"
            )

            usd_inr = float(
                exchange["rates"]["INR"]
            )

            grams_per_ounce = 31.1034768

            inr_per_gram = (
                price_usd
                * usd_inr
                / grams_per_ounce
            )

            return (
                "Live silver spot price:\n"
                f"₹{inr_per_gram:,.2f} per gram\n"
                f"${price_usd:,.2f} per troy ounce\n"
                f"USD/INR: ₹{usd_inr:,.2f}"
            )

        except Exception as error:

            print(
                "[LiveData] Silver API error:",
                error
            )

            return None

    # =========================================================
    # DETECT LIVE PRICE QUESTION
    # =========================================================

    def get_price(self, query):

        q = query.lower().strip()

        # GOLD
        if (
            "gold" in q
            and any(
                word in q
                for word in [
                    "price",
                    "rate",
                    "cost",
                    "value"
                ]
            )
        ):

            return self.gold_price()

        # SILVER
        if (
            "silver" in q
            and any(
                word in q
                for word in [
                    "price",
                    "rate",
                    "cost",
                    "value"
                ]
            )
        ):

            return self.silver_price()

        return None
