
import { NuxtPage } from '~/.nuxt/components';

<template>
    <div class="panel right">

<NuxtPage :FormData="FormData"></NuxtPage>
    </div>


</template>

<style scoped>

.panel {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 100%;
    width: 100%;
    background-color: #1e1e2e;
    position: relative;
    border: 1px solid #28283E;
   
}

</style>

<script>

export default {
    name: 'RightPanel',
    props: {
        FormData: Object,
    },
}

</script>