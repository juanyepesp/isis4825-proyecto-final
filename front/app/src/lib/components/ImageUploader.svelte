<script lang="ts">
  export let file: File | null;
  export let onFileSelect: (file: File | null) => void;
</script>

<label class="upload-box">
  <input
    type="file"
    accept="image/*"
    on:change={(e) => {
      const files = (e.target as HTMLInputElement).files;
      onFileSelect(files?.[0] ?? null);
    }}
  />
  {#if file}
    <span>{file.name}</span>
  {:else}
    <span>Haga click aquí para escoger la imagen</span>
  {/if}
</label>

<style>
  .upload-box {
    border: 2px dashed #ccc;
    padding: 2rem;
    cursor: pointer;
    border-radius: 10px;
    background: #f9f9f9;
    display: block;
    width: 100%;
    text-align: center;
    margin-bottom: 1rem;
  }
  .upload-box input {
    display: none;
  }
</style>