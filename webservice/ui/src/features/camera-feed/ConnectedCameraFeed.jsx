import { connect } from "react-redux";
import CameraFeed from "./CameraFeed";
import { videoOn } from "../../dux/stats";

// maps the redux state to this components props
const mapStateToProps = state => ( {
  videoOn: state.stats.videoOn,
} );

// provide the component with the dispatch method
const mapDispatchToProps = dispatch => ( {

} );

export default connect( mapStateToProps, mapDispatchToProps )(  CameraFeed);
