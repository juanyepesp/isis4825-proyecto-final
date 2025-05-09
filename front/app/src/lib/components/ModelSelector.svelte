<script lang="ts">
  export let models: string[];
  export let selectedModel: string;
  export let onSelect: (model: string) => void;
</script>

<div class="model-selector">
  {#each models as model}
    <button
      class="model-button"
      class:selected={model === selectedModel}
      on:click={() => onSelect(model)}
    >
      {model}
    </button>
  {/each}
</div>

<style>
  .model-selector {
    display: flex;
    justify-content: center;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-bottom: 1rem;
  }
  .model-button {
    padding: 0.5rem 1rem;
    border: 2px solid #ccc;
    background: white;
    cursor: pointer;
    border-radius: 8px;
  }
  .model-button.selected {
    background-color: #007bff;
    color: white;
    border-color: #007bff;
  }
</style>