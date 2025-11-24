from quart import Quart, request, make_response
from elasticmm.engine.pipeline import create_disagg_proxy_app


# For simplicity, we wrap pipeline's app with configurable fanout using env or defaults.

def create_api_server(fanout_prefill: int = 1, fanout_decode: int = 1) -> Quart:
	# service discovery and proxy binding are fixed here; can be env-driven
	app = create_disagg_proxy_app(
		service_discovery_host="0.0.0.0",
		service_discovery_port=30002,
		api_host="0.0.0.0",
		api_port=10001,
		fanout_prefill=fanout_prefill,
		fanout_decode=fanout_decode,
	)
	return app


if __name__ == "__main__":
	app = create_api_server()
	app.run(host="0.0.0.0", port=10001)