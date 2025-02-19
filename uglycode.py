import ray
from eformer.escale.tpexec import TPUMultiSliceExecutor

if __name__ == "__main__":
	ray.init("auto")

	@ray.remote
	def fn():
		import jax
		import transformers
		from jax import numpy as jnp
		from jax.sharding import PartitionSpec as PS
		import easydel as ed
		from eformer.escale import with_sharding_constraint

		pretrained_model_name_or_path = "meta-llama/Llama-3.2-1B-Instruct"
		tokenizer = transformers.AutoTokenizer.from_pretrained(
			pretrained_model_name_or_path
		)
		tokenizer.padding_side = "left"
		tokenizer.pad_token_id = tokenizer.eos_token_id
		print("Loading", pretrained_model_name_or_path)
		model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
			pretrained_model_name_or_path,
			auto_shard_model=True,
			sharding_axis_dims=(2, 1, 1, -1),
			config_kwargs=ed.EasyDeLBaseConfigDict(
				freq_max_position_embeddings=8192,
				mask_max_position_embeddings=8192,
				kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.NONE,
				attn_dtype=jnp.bfloat16,
				attn_softmax_dtype=jnp.float32,
				attn_mechanism=ed.AttentionMechanisms.VANILLA,
			),
			quantization_method=ed.EasyDeLQuantizationMethods.A8BIT,
			param_dtype=jnp.bfloat16,
			dtype=jnp.bfloat16,
			precision=jax.lax.Precision("fastest"),
		)
		inference = ed.vInference(
			model=model,
			processor_class=tokenizer,
			generation_config=ed.vInferenceConfig(
				max_new_tokens=1024,
				temperature=0.0,
				do_sample=False,
				top_p=0.95,
				top_k=10,
				eos_token_id=model.generation_config.eos_token_id,
				streaming_chunks=32,
				num_return_sequences=2,
			),
		)

		print(model.model_task)
		print(model.model_type)
		inference.precompile(2, [2048])

		messages = [
			{
				"role": "system",
				"content": "You are a helpful AI assistant.",
			},
			{
				"role": "user",
				"content": "Can you provide ways to eat combinations of bananas and dragonfruits?",
			},
			{
				"role": "assistant",
				"content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey.",
			},
			{
				"role": "user",
				"content": "What about solving an 2x + 3 = 7 equation?",
			},
		]
		ids = tokenizer.apply_chat_template(
			messages,
			return_tensors="jax",
			return_dict=True,
			add_generation_prompt=True,
		)

		print("Start Generation Process.")
		import numpy

		for i, response in enumerate(inference.generate(**ids)):
			...
		print("generated")
		with model.mesh:
			taken = with_sharding_constraint(
				response.sequences[..., response.padded_length :],
				PS(),
			)
		return numpy.array(taken)

	features = TPUMultiSliceExecutor.execute(fn, "v3-8", 2)
	print(features)
	results = ray.get(features[0])
	print(results.result)
        ray.shutdown()
