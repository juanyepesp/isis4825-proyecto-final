<script lang="ts">
  export let originalFile: File | null;
  export let predictionUrl: string | null;
</script>

{#if originalFile && predictionUrl}
  <div class="result-grid">
    <div>
      <h3>Original Image</h3>
      <img src={URL.createObjectURL(originalFile)} alt="Uploaded" />
    </div>
    <div>
      <h3>Prediction Result</h3>
      <img src={predictionUrl} alt="Prediction" />
    </div>
  </div>
{/if}

<style>
  .result-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
    align-items: start;
  }
  .result-grid img {
    max-width: 100%;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
  }
  @media (max-width: 600px) {
    .result-grid {
      grid-template-columns: 1fr;
    }
  }
</style>