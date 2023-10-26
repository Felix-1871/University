<template>
 <InterfaceNavbar/>
<div class="PanelContent"> 
<LeftPanel :FormData="FormData"/>

<RightPanel :FormData="FormData"/></div> 
</template>

<style scoped>

.PanelContent {
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    height: 100%;
    background-color: #1e1e1e;
    overflow: hidden;
}


</style>

<script>
import LeftPanel from "./LeftPanel.vue";
import RightPanel from "./RightPanel.vue";

console.log(FormData)
export default {
    name: 'MainInterface',
    props: {
        FormData: Object,
    },
    components: { LeftPanel, RightPanel }
}



</script>