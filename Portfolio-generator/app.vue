<template>
<div class="SiteContent">
 <Form></Form>
</div>

</template>

<script>

</script>

<style>
@import "@catppuccin/palette/style";
html, body {
    height: 100vh;
    width: 100vw;
    padding: 0;
    background-color: #000;
    background-image: radial-gradient(circle at top right, rgba(121, 68, 154, 0.13), transparent), radial-gradient(circle at 20% 80%, #101019, transparent);
    font-family: 'Montserrat', sans-serif;
    overflow: hidden;
    color: var(--ctp-mocha-text);

}
.SiteContent {
    height: 100vh;
    width: 100vw;
}
.interface {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 400px;
    width: 810px;
    position: absolute; 
        left: 0;
        right: 0;
        top: 0;
        bottom: 0;
        margin: auto;
        max-width: 100%;
        max-height: 100%;

        box-shadow: 0 0 10px rgba(255, 255, 255, 0.5);

}


</style>

<script>
</script>