# Copyright (c) 2025-2026 Sunet.
# Contributor: Kristofer Hallin
#
# This file is part of Sunet Scribe.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

from typing import Optional

import requests


def get_token() -> Optional[str]:
    try:
        auth_data = {
            "grant_type": "client_credentials",
            "client_id": os.environ["OIDC_CLIENT_ID"],
            "client_secret": os.environ["OIDC_CLIENT_SECRET"],
        }
        auth_response = requests.post(
            os.environ["OIDC_TOKEN_ENDPOINT"],
            data=auth_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=5,
        )
        auth_response.raise_for_status()
        token = auth_response.json()["access_token"]
        return token
    except Exception as e:
        raise ValueError("Could not get JWT token: {}".format(e))
        return None


if __name__ == "__main__":
    args_failure = False

    if "OIDC_TOKEN_ENDPOINT" not in os.environ:
        print("OIDC_TOKEN_ENDPOINT not set")
        args_failure = True
    if "OIDC_CLIENT_ID" not in os.environ:
        print("OIDC_CLIENT_ID not set")
        args_failure = True

    if "OIDC_CLIENT_SECRET" not in os.environ:
        print("OIDC_CLIENT_SECRET not set")
        args_failure = True

    if args_failure:
        sys.exit(1)

    print(get_token())
