<template>
  <div class="SiteInterface">
    <MainContainer />
  </div>
</template>

<style>

@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');
@import "@catppuccin/palette/style";

.SiteInterface {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100vh;
}

html, body {

  font-family: 'Montserrat', sans-serif;
  background-color: var(--ctp-mocha-base);
  color: var(--ctp-mocha-text);
}

</style>
