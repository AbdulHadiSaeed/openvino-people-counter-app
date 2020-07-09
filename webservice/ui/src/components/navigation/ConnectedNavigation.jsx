import { connect } from "react-redux";
import Navigation from "./Navigation";
import { toggleStats, toggleVideo } from "../../dux/stats";

// maps the redux state to this components props
const mapStateToProps = state => ( {
  statsOn: state.stats.statsOn,
  videoOn: state.stats.videoOn,
} );

// provide the component with the dispatch method
const mapDispatchToProps = dispatch => ( {
  toggleStats: () => {
    dispatch( toggleStats() );
  },
  toggleVideo: () => {
    dispatch( toggleVideo() );
  },
} );

export default connect( mapStateToProps, mapDispatchToProps )( Navigation );
